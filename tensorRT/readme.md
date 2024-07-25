
# 安装

    python3 -m pip install --upgrade tensorrt

# plugin解释
   ##  Creator
    ### 父类nvinfer1::IPluginCreator
    在NvInferRuntimeCommon.h 文件中
   * 是一个纯虚类，方法都要在子类中实现，方法必须override,没有自己的方法
### 数据类：PluginFieldCollection 

 * 存放参数列表的开头指针，以及参数个数
    struct PluginFieldCollection
    
    {
        int32_t nbFields;          //!参数个数
        const PluginField* fields; //!<指针
     }

### 数据类：PluginField
* 一个属性attr类，保存属性的name,属性的data，属性的数据类型，属性的指针，属性的长度

### std::vector   <nvinfer1::PluginField> 
 * 真实保存属性数据的列表，所有属性都在里面

* creator 的作用和onnxruntime里面的一样，是想传入参数后，返回实例化的kernel
* creator会将属性保存成列表的形式，自身保存了命名空间和layer的名字
## 父类 IPluginV2

   在NvInferRuntimeCommon.h 文件中
   方法就是现在写在kernel内的所有方法
   是一个纯虚类，所有方法都要在子类中重写
   
## 子类 IPluginV2Ext  

   在这里实现了两个IPluginV2中的方法，其他的增加函数依然是纯虚函数
  
## 孙子类IPluginV2DynamicExt
   在这里实现了 IPluginV2Ext中的一些方法，其他的都是纯虚函数
# 数据：DimsExprs
    只负责表示张量的形状，是个数组
    class DimsExprs
    {
    public:
        int32_t nbDims;                          //!< The number of dimensions.
        const IDimensionExpr* d[Dims::MAX_DIMS]; //!< The extent of each dimension.
    };
    class IDimensionExpr
    {
    public:
        virtual bool isConstant() const = 0;
        virtual int32_t getConstantValue() const = 0;
    protected:
        virtual ~IDimensionExpr() {}
    };

# IExprBuilder

     class IExprBuilder
    {
    public:
        virtual const IDimensionExpr* constant(int32_t value) = 0;
        virtual const IDimensionExpr* operation(DimensionOperation op, const IDimensionExpr& first, const IDimensionExpr& second) = 0;

    protected:
        virtual ~IExprBuilder() {}
    };
    
# 结构 PluginTensorDesc
    张量的形状信息
        class Dims
        {
        public:
            static const int32_t MAX_DIMS = 8;  
            int32_t nbDims;  //!< The number of dimensions.
            int32_t d[MAX_DIMS];  //!< The extent of each dimension.
        };
        
        主要的，张量的相关属性，包括形状，数据类型，
        struct PluginTensorDesc
        {
            Dims dims;   //最重要的
            DataType type; //!< \warning DataType:kBOOL not supported.
            TensorFormat format;
            float scale;
        };
        
        动态张量属性
        struct DynamicPluginTensorDesc
        {
            PluginTensorDesc desc;
            Dims min;
            Dims max;
        };
            
        数据类型,使用两次，用于选择合适的数据格式匹配
        enum class DataType : int32_t
        {
            kFLOAT = 0,
            kHALF = 1,
            kINT8 = 2,
            kINT32 = 3,
            kBOOL = 4
        };

 # 以上用到的数据类型都是表示张量的属性类，主要负责类型和形状
    ctypes.CDLL(osp.join(dir_path, 'libamirstan_plugin.so'))
 #### 两个关键类
    属性列表
    struct PluginFieldCollection
    {
        int32_t nbFields;          //属性个数
        const PluginField* fields; //属性列表开头指针
    };
    一个属性（name,data,type,length）
    class PluginField
    {
    public:
        const char* name{nullptr};
        const void* data{nullptr};
        PluginFieldType type{PluginFieldType::kUNKNOWN};
        int32_t length{0};
        PluginField(const char* name_ = nullptr, 
                const void* data_ = nullptr, 
                const PluginFieldType type_ = PluginFieldType::kUNKNOWN,
                int32_t length_ = 0): 
        name(name_), data(data_), type(type_), length(length_){}
    };
    #创建一个creator，开始构造一个类creator 
    creator = trt.get_plugin_registry().get_plugin_creator('DeformablePoolPluginDynamic', '1', '')
    #初始化就是创建存放属性的列表
