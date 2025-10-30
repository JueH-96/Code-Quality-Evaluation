# YOUR CODE HERE
def mex(a, b, c):
    s = set([a, b, c])
    for i in range(3):
        if i not in s:
            return i
    return 3

N = int(input())
A = list(map(int, input().split()))
S = input()

m_indices = [i for i, c in enumerate(S) if c == 'M']
e_indices = [i for i, c in enumerate(S) if c == 'E']
x_indices = [i for i, c in enumerate(S) if c == 'X']

total = 0

for i in m_indices:
    for j in e_indices:
        if j > i:
            for k in x_indices:
                if k > j:
                    total += mex(A[i], A[j], A[k])

print(total)