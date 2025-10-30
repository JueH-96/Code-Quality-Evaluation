# T = int(input())
# for _ in range(T):
#     S = input()
#     X = input()
#     Y = input()
#     i = 0
#     j = 0
#     while i < len(X) and j < len(Y):
#         if X[i] != Y[j]:
#             break
#         if X[i] == '0':
#             i += 1
#         else:
#             j += 1
#     if i == len(X) and j == len(Y):
#         print('Yes')
#     elif i == len(X) or j == len(Y):
#         print('No')
#     else:
#         if S == '' or S[0] == X[i] and S[0] == Y[j]:
#             print('Yes')
#         else:
#             print('No')
#     while i < len(X) and X[i] == '0':
#         i += 1
#     while j < len(Y) and Y[j] == '0':
#         j += 1
#     if i == len(X) and j == len(Y):
#         print('Yes')
#     elif i == len(X) or j == len(Y):
#         print('No')
#     else:
#         if S == '' or S[0] == X[i] and S[0] == Y[j]:
#             print('Yes')
#         else:
#             print('No')
# else:
#     print('Yes')