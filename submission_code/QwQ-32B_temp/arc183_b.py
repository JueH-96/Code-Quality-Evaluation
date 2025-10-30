import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N, K = map(int, input[ptr:ptr+2])
        ptr += 2
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        B = list(map(int, input[ptr:ptr+N]))
        ptr += N

        left = 0
        right = 0
        freq = defaultdict(int)
        freq[A[0]] = 1
        possible = True

        for i in range(N):
            desired_left = max(0, i - K)
            desired_right = min(N-1, i + K)

            # Expand to the right
            while right < desired_right:
                right += 1
                freq[A[right]] += 1

            # Expand to the left (if needed)
            while left > desired_left:
                left -= 1
                freq[A[left]] += 1

            # Contract to the left (if needed)
            while left < desired_left:
                freq[A[left]] -= 1
                if freq[A[left]] == 0:
                    del freq[A[left]]
                left += 1

            if B[i] not in freq:
                possible = False
                break

        print("Yes" if possible else "No")

if __name__ == "__main__":
    main()