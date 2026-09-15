import torch

# 加载模型文件为状态字典
state_dict = torch.load('work_dirs/drivestudio/omnire_23/checkpoint_final.pth', map_location='cpu')


# 查看顶层的所有键
# print(state_dict.keys())

# # 如果是 state_dict，查看各层的名字
# if isinstance(state_dict, dict):
#     for key in state_dict.keys():
#         print(f"Key: {key}")

#print(state_dict['models'].keys())# 模型键值，包含高斯场景图各节点名
#print(f"Background keys: {state_dict['models']['Background'].keys()}")# Background节点键值，包含高斯分布数据和配置参数等信息
# print(f"RigidNodes means: {state_dict['models']['RigidNodes']['_means']}")
#print(f"DeformableNodes keys: {state_dict['models']['DeformableNodes'].keys()}")
#print(f"SMPLNodes keys: {state_dict['models']['SMPLNodes'].keys()}")
#print(f"Sky keys: {state_dict['models']['Sky'].keys()}")
