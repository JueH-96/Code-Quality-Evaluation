def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    numbers = list(map(int, input_data))
    
    # First 9 numbers are Takahashi's scores and next 8 numbers are Aoki's scores
    takahashi_scores = numbers[:9]
    aoki_scores = numbers[9:17]
    
    total_takahashi = sum(takahashi_scores)
    total_aoki = sum(aoki_scores)
    
    # Aoki needs to win so he must score strictly more than Takahashi's total.
    # Thus, he needs to beat him by at least one run.
    runs_needed = total_takahashi - total_aoki + 1
    
    print(runs_needed)

if __name__ == '__main__':
    main()