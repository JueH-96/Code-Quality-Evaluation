import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = map(int, data[1:])
    
    prefix_sum = 0
    min_prefix = 0
    for a in A:
        prefix_sum += a
        if prefix_sum < min_prefix:
            min_prefix = prefix_sum
    
    # Initial passengers needed so that prefix never dips below 0
    initial = -min_prefix  # note min_prefix is <= 0
    # Final passengers is initial plus total sum
    result = initial + prefix_sum
    print(result)

if __name__ == "__main__":
    main()