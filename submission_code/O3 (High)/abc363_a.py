import sys

def main() -> None:
    R = int(sys.stdin.readline().strip())
    
    # Next threshold at the next multiple of 100
    next_threshold = (R // 100 + 1) * 100
    increment_needed = next_threshold - R
    
    print(increment_needed)

if __name__ == "__main__":
    main()