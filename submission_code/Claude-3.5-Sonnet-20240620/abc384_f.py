# YOUR CODE HERE
def f(x):
    while x % 2 == 0:
        x //= 2
    return x

def solve(N, A):
    freq = {}
    for a in A:
        fa = f(a)
        freq[fa] = freq.get(fa, 0) + 1
    
    result = 0
    for i, a in enumerate(A):
        for b in A[i:]:
            result += f(a + b)
    
    return result

N = int(input())
A = list(map(int, input().split()))

print(solve(N, A))