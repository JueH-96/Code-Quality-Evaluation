# N = int(input())
# A = list(map(int, input().split()))

def max_experience(N, A):
    even = 0
    odd = 0
    dp_even = [0] * (N + 1)
    dp_odd = [0] * (N + 1)
    
    for i in range(N):
        if i % 2 == 0:
            dp_even[i + 1] = max(dp_even[i], dp_odd[i] + A[i])
            dp_odd[i + 1] = max(dp_odd[i], dp_even[i])
            even += A[i]
        else:
            dp_odd[i + 1] = max(dp_odd[i], dp_even[i] + A[i])
            dp_even[i + 1] = max(dp_even[i], dp_odd[i])
            odd += A[i]
    
    return max(dp_even[N], dp_odd[N])

N = int(input())
A = list(map(int, input().split()))
print(max_experience(N, A))