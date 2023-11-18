import torch

# PATH = ''
# def save_classifier(model, method, context, index):
#     # if method == 'none':
#     #     torch.save(model.state_dict(),
#     #                '{}/baseline/{}_classifier_{}_context{}.pt'.format(PATH, method, index, context))
#     # elif method == 'joint':
#     #     torch.save(model.state_dict(),
#     #                '{}/baseline/{}_classifier_{}_context{}.pt'.format(PATH, method, index, context))
#     # elif method == 'icarl':
#     #     torch.save(model,
#     #                '{}/{}/{}_classifier_{}_context{}.pt'.format(PATH, method, index, method, context))
#     # else:
#     #     torch.save(model.state_dict(),
#     #                '{}/{}/{}_classifier_{}_context{}.pt'.format(PATH, method, index, method, context))
#     if context == 10:
#         if method == 'none':
#             torch.save(model.state_dict(),
#                        '{}/baseline/{}_classifier_{}_context{}.pt'.format(PATH, method, index, context))
#         elif method == 'joint':
#             torch.save(model.state_dict(),
#                        '{}/baseline/{}_classifier_{}_context{}.pt'.format(PATH, method, index, context))
#         elif method == 'icarl':
#             torch.save(model,
#                        '{}/{}/{}_classifier_{}_context{}.pt'.format(PATH, method, index, method, context))
#         else:
#             torch.save(model.state_dict(),
#                        '{}/{}/{}_classifier_{}_context{}.pt'.format(PATH, method, index, method, context))
#     return 0

PATH = ''
def save_classifier(model, method, context, index):
    if context == 10:
        if method == 'none':
            torch.save(model.state_dict(),
                       '{}/{}_classifier_{}_context{}.pt'.format(PATH, method, index, context))
        elif method == 'joint':
            torch.save(model.state_dict(),
                       '{}/{}_classifier_{}_context{}.pt'.format(PATH, method, index, context))
        elif method == 'icarl':
            torch.save(model,
                       '{}/{}_classifier_{}_context{}.pt'.format(PATH, index, method, context))
        else:
            torch.save(model.state_dict(),
                       '{}/{}_classifier_{}_context{}.pt'.format(PATH, index, method, context))
    # if method == 'none':
    #     torch.save(model.state_dict(),
    #                '{}/baseline/{}_classifier_{}_context{}.pt'.format(PATH, method, index, context))
    # elif method == 'joint':
    #     torch.save(model.state_dict(),
    #                '{}/baseline/{}_classifier_{}_context{}.pt'.format(PATH, method, index, context))
    # else:
    #     torch.save(model.state_dict(),
    #                '{}/{}/{}_classifier_{}_context{}.pt'.format(PATH, method, index, method, context))
    return 0

def save_gan(gen, disc, method, context, index):
    if context == 10:
        torch.save(gen.state_dict(),
                   '{}/{}/{}_gen_{}_context{}.pt'.format(PATH, method, index, method, context))
        torch.save(disc.state_dict(),
                   '{}/{}/{}_disc_{}_context{}.pt'.format(PATH, method, index, method, context))
    return 0
