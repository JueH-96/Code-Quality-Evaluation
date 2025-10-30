# YOUR CODE HERE
from sys import stdin, stdout

def main():
    N, Q = map(int, stdin.readline().split())
    H = list(map(int, stdin.readline().split()))

    # Precompute the next greater element for each building
    stack = []
    next_greater = [N] * N
    for i in range(N - 1, -1, -1):
        while stack and H[stack[-1]] < H[i]:
            stack.pop()
        if stack:
            next_greater[i] = stack[-1]
        stack.append(i)

    # Process queries
    for _ in range(Q):
        l, r = map(int, stdin.readline().split())
        l -= 1
        r -= 1
        
        # Find the tallest building between l and r
        max_height = max(H[l], H[r])
        max_index = l if H[l] > H[r] else r

        # Count visible buildings
        count = 0
        current = r + 1
        while current < N and H[current] < max_height:
            if H[current] > H[r]:
                count += 1
            current = next_greater[current]

        stdout.write(f"{count}
")

if __name__ == "__main__":
    main()