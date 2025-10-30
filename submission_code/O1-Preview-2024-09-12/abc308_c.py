# YOUR CODE HERE
N = int(input())
A_list = []
for _ in range(N):
    A_i_str, B_i_str = input().split()
    A_i = int(A_i_str)
    B_i = int(B_i_str)
    A_list.append((A_i, B_i))

indices = list(range(N))

def key_func(i):
    A_i, B_i = A_list[i]
    return (-A_i / (A_i + B_i), i + 1)

sorted_indices = sorted(indices, key=key_func)
result = [str(idx + 1) for idx in sorted_indices]
print(' '.join(result))