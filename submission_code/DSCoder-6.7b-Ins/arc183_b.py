import sys

def solve():
    T = int(sys.stdin.readline().strip())
    for _ in range(T):
        N, K = map(int, sys.stdin.readline().strip().split())
        A = list(map(int, sys.stdin.readline().strip().split()))
        B = list(map(int, sys.stdin.readline().strip().split()))

        # Create a dictionary to store the positions of each number in A
        pos = {}
        for i in range(N):
            if A[i] not in pos:
                pos[A[i]] = []
            pos[A[i]].append(i)

        # Try to match B[i] with A[i] or A[i+K]
        for i in range(N):
            if B[i] in pos:
                # If B[i] is in A, try to match it with A[i] or A[i+K]
                for j in pos[B[i]]:
                    if j <= i + K:
                        # If A[j] is within K distance from A[i], replace A[i] with A[j]
                        A[i] = B[i]
                        break
            else:
                # If B[i] is not in A, it's impossible to make A identical to B
                print("No")
                return

        # If we've made it through all the numbers in B, A is identical to B
        print("Yes")

# Call the function to solve the problem
solve()