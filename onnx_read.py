import numpy as np
import onnx

ONNX_PATH = "./models/shufflenet_v2_x0_5-half.onnx"

model = onnx.load(ONNX_PATH)

print(model.doc_string)

print(model.graph.input)
# [name: "image"
# type {
#   tensor_type {
#     elem_type: 1
#     shape {
#       dim {
#         dim_value: 1
#       }
#       dim {
#         dim_value: 3
#       }
#       dim {
#         dim_value: 224
#       }
#       dim {
#         dim_value: 224
#       }
#     }
#   }
# }
# ]
print(model.graph.input[0].name)
# image
print(model.graph.input[0].type)
# tensor_type {
#   elem_type: 1
#   shape {
#     dim {
#       dim_value: 1
#     }
#     dim {
#       dim_value: 3
#     }
#     dim {
#       dim_value: 224
#     }
#     dim {
#       dim_value: 224
#     }
#   }
# }
print(model.graph.input[0].type.tensor_type)
# shape {
#   dim {
#     dim_value: 1
#   }
#   dim {
#     dim_value: 3
#   }
#   dim {
#     dim_value: 224
#   }
#   dim {
#     dim_value: 224
#   }
# }
print(model.graph.input[0].type.tensor_type.shape)
# dim {
#   dim_value: 1
# }
# dim {
#   dim_value: 3
# }
# dim {
#   dim_value: 224
# }
# dim {
#   dim_value: 224
# }
print(model.graph.input[0].type.tensor_type.shape.dim)
# [ dim_value: 1
# , dim_value: 3
# , dim_value: 224
# , dim_value: 224
# ]
print(model.graph.input[0].type.tensor_type.shape.dim[0])
# dim_value: 1
print(model.graph.input[0].type.tensor_type.shape.dim[0])

# print(model.graph.output)
# [name: "classes"
# type {
#   tensor_type {
#     elem_type: 1
#     shape {
#       dim {
#         dim_value: 1
#       }
#       dim {
#         dim_value: 1000
#       }
#     }
#   }
# }
# ]
