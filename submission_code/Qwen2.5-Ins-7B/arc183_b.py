# T = int(input())
# for _ in range(T):
#     N, K = map(int, input().split())
#     A = list(map(int, input().split()))
#     B = list(map(int, input().split()))
#     flag = True
#     for i in range(N):
#         if A[i] != B[i]:
#             for j in range(max(0, i-K), min(N, i+K+1)):
#                 if A[j] == B[i]:
#                     A[i] = B[i]
#                     break
#             else:
#                 flag = False
#                 break
#     print('Yes' if flag else 'No')
# YOUR CODE HERE
T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    freq_A = [0] * (N + 1)
    freq_B = [0] * (N + 1)
    for i in range(N):
        freq_A[A[i]] += 1
        freq_B[B[i]] += 1
    for i in range(1, N + 1):
        if freq_A[i] < freq_B[i]:
            print('No')
            break
    else:
        print('Yes')