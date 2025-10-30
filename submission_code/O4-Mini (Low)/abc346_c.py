import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    A = map(int, data[2:])
    
    # Total sum of 1..K
    total = K * (K + 1) // 2
    
    # Sum of distinct A_i that lie between 1 and K
    seen = set()
    present_sum = 0
    for x in A:
        if x <= K and x not in seen:
            seen.add(x)
            present_sum += x
    
    # The answer is total minus the sum of present numbers
    print(total - present_sum)

if __name__ == "__main__":
    main()