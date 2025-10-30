# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    queries = []
    for i in range(Q):
        l = int(data[3 + 2*i])
        r = int(data[4 + 2*i])
        queries.append((l, r))
    
    # Preprocessing: find all positions where S[p] == S[p+1]
    consecutive = []
    for p in range(N-1):
        if S[p] == S[p+1]:
            consecutive.append(p+1)  # 1-based index
    
    # For each query, count the number of positions in consecutive that are between l and r-1
    for l, r in queries:
        # Convert to 1-based index
        # We need to find the number of p where l <= p <= r-1 and p is in consecutive
        # Since consecutive is sorted, we can use binary search
        low = 0
        high = len(consecutive) - 1
        left = -1
        right = -1
        # Find the first p >= l
        while low <= high:
            mid = (low + high) // 2
            if consecutive[mid] >= l:
                high = mid - 1
            else:
                low = mid + 1
        left = low
        # Find the last p <= r-1
        low2 = 0
        high2 = len(consecutive) - 1
        while low2 <= high2:
            mid2 = (low2 + high2) // 2
            if consecutive[mid2] <= r-1:
                low2 = mid2 + 1
            else:
                high2 = mid2 - 1
        right = high2
        # The count is right - left + 1
        if left > right:
            print(0)
        else:
            print(right - left + 1)

if __name__ == "__main__":
    main()