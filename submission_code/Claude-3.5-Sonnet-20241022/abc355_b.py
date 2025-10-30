N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Combine and sort all elements
C = sorted(A + B)

# Check if any consecutive elements are both from A
for i in range(len(C)-1):
    if C[i] in A and C[i+1] in A:
        print("Yes")
        exit()

print("No")