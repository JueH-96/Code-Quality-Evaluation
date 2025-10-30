# YOUR CODE HERE
import sys
input = sys.stdin.read

def can_color_all_points(N, K):
    if N % 2 == 0 and K % (N // 2) == 0:
        return "No"
    return "Yes"

def main():
    data = input().split()
    T = int(data[0])
    results = []
    index = 1
    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        results.append(can_color_all_points(N, K))
        index += 2
    print("
".join(results))

if __name__ == "__main__":
    main()