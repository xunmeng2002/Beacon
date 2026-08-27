// Python 绑定：导出门面 VectorDb + Metric + Hit 为模块 beacon（import beacon）
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Beacon/VectorDb.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace beacon;

namespace
{
// 将 Python 入参（list 或 numpy 数组）转为一维 float32 向量；forcecast 自动做类型/形状转换
std::vector<float> ToVector(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr,
                            const char* what)
{
    const py::buffer_info info = arr.request();
    if (info.ndim != 1)
    {
        throw py::value_error(std::string(what) + " 须为 1 维数组，实际维度 " + std::to_string(info.ndim));
    }
    std::vector<float> result;
    const auto n = static_cast<std::size_t>(info.shape[0]);
    result.reserve(n);
    if (n > 0)
    {
        const auto* ptr = static_cast<const float*>(info.ptr);
        result.assign(ptr, ptr + n);
    }
    return result;
}

// 重操作（HNSW 搜索）释放 GIL，避免长查询阻塞 Python 其他线程；VectorDb 内部共享读锁，线程安全
std::vector<Hit> SearchExactImpl(const VectorDb& db,
                                 const py::array_t<float, py::array::c_style | py::array::forcecast>& query,
                                 std::size_t k)
{
    const std::vector<float> q = ToVector(query, "query");
    py::gil_scoped_release release;
    return db.SearchExact(q, k);
}

std::vector<Hit> SearchIndexedImpl(const VectorDb& db,
                                   const py::array_t<float, py::array::c_style | py::array::forcecast>& query,
                                   std::size_t k, std::size_t ef)
{
    const std::vector<float> q = ToVector(query, "query");
    py::gil_scoped_release release;
    return db.SearchIndexed(q, k, ef);
}
}  // namespace

PYBIND11_MODULE(beacon, m)
{
    m.doc() = "Beacon：自研 C++20 内存向量数据库（HNSW 近似检索 + 精确 Top-K）";

    py::enum_<Metric>(m, "Metric")
        .value("kCosine", Metric::kCosine)
        .value("kInnerProduct", Metric::kInnerProduct);

    py::class_<Hit>(m, "Hit")
        .def_readonly("id", &Hit::id)
        .def_readonly("score", &Hit::score)
        .def("__repr__", [](const Hit& h)
        {
            return "<Hit id=" + std::to_string(h.id) + " score=" + std::to_string(h.score) + ">";
        });

    py::class_<VectorDb>(m, "VectorDb")
        .def(py::init<std::size_t, Metric>(), py::arg("dim"), py::arg("metric") = Metric::kCosine)
        .def("add",
             [](VectorDb& db, const py::array_t<float, py::array::c_style | py::array::forcecast>& vec,
                const std::string& meta)
             {
                 const std::vector<float> v = ToVector(vec, "vector");
                 return db.Add(v, meta);
             },
             py::arg("vector"), py::arg("meta") = "")
        .def("update",
             [](VectorDb& db, std::size_t id,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& vec,
                const std::string& meta)
             {
                 const std::vector<float> v = ToVector(vec, "vector");
                 return db.Update(id, v, meta);
             },
             py::arg("id"), py::arg("vector"), py::arg("meta") = "")
        .def("delete", &VectorDb::Delete, py::arg("id"))
        .def("search_exact", &SearchExactImpl, py::arg("query"), py::arg("k"))
        .def("reserve", &VectorDb::Reserve, py::arg("slot_count"))
        .def("enable_index", &VectorDb::EnableIndex, py::arg("m") = 16, py::arg("ef_construction") = 200)
        .def("disable_index", &VectorDb::DisableIndex)
        .def("index_enabled", &VectorDb::IndexEnabled)
        .def("search_indexed", &SearchIndexedImpl, py::arg("query"), py::arg("k"), py::arg("ef") = 100)
        .def_property_readonly("count", &VectorDb::count)
        .def_property_readonly("dim", &VectorDb::dim)
        .def("deleted", &VectorDb::deleted, py::arg("id"))
        .def("metadata", &VectorDb::metadata, py::arg("id"))
        .def("save", &VectorDb::Save, py::arg("path"))
        .def("load", &VectorDb::Load, py::arg("path"))
        .def("clear", &VectorDb::Clear)
        .def("__len__", &VectorDb::count);
}
