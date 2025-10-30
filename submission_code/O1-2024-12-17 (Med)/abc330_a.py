def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, L = map(int, data[:2])
    scores = list(map(int, data[2:]))

    # Count how many scores are at least L
    count_passed = sum(1 for score in scores if score >= L)
    
    print(count_passed)

# Do not forget to call the main function
main()