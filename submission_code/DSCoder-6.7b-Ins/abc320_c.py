import sys

def read_input():
    M = int(sys.stdin.readline().strip())
    S = [sys.stdin.readline().strip() for _ in range(3)]
    return M, S

def solve(M, S):
    # Check if all reels have the same symbols
    if len(set(S)) == 1:
        return 0

    # Check if it's impossible to make all reels display the same character
    if all(s[0] != s[-1] for s in S):
        return -1

    # Calculate the minimum time to stop all reels
    time = 0
    for i in range(M):
        # Calculate the time to stop each reel
        t = min(s[i] != S[j][0] for j, s in enumerate(S))
        time += t

    return time

def main():
    M, S = read_input()
    print(solve(M, S))

if __name__ == "__main__":
    main()