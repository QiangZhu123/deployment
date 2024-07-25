// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//! [op:common_include]
#include <openvino/op/op.hpp>
//! [op:common_include]

//! [op:header]
namespace TemplateExtension {
//继承的就是一个Node类
class Identity : public ov::op::Op {
public:
    //这是一个宏，定义他的名字，版本等信息，NodeTypeInfo ,
    //OPENVINO_OP("name","version")或者是("name",int) 
    //或者是static constexpr NodeTypeInfo type_info{"Acos", 0};
    //const NodeTypeInfo& get_type_info() const override { return type_info; }
    OPENVINO_OP("Identity");
    
    //默认构造函数，
    Identity() = default;
    
    //有参数的构造函数，有多少属性参数都要写进去
    Identity(const ov::Output<ov::Node>& arg);
    
    //调用多次，来推断出输出的形状和类型
    //会使用输入的形状和类型 ，
    //使用Output的方法获得输入形状get_input_partial_shape() 和元素类型get_input_element_type()
    //设置输出的形状和类型 用set_output_type.
    //父类中什么都没做，可能没有
    void validate_and_infer_types() override;
    
    //创建复制，返回参数的智能指针，
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    
    //用于序列化和反序列化属性，用visiter.on_attribute方法遍历所有的属性
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    
    //可选，计算函数
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
//! [op:header]

}  // namespace TemplateExtension
