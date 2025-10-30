import sys

def main():
    n = int(sys.stdin.readline())
    a_list = list(map(int, sys.stdin.readline().split()))
    
    current_sum = 0
    min_prefix = 0  # Initialize with S_0 = 0
    
    for a in a_list:
        current_sum += a
        if current_sum < min_prefix:
            min_prefix = current_sum
    
    required_x = max(0, -min_prefix)
    result = required_x + current_sum
    print(result)

if __name__ == "__main__":
    main()