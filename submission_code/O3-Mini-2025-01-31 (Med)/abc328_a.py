def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # First two elements: N and X
    N, X = int(input_data[0]), int(input_data[1])
    
    # Rest elements: scores S[1] to S[N]
    scores = list(map(int, input_data[2:2+N]))
    
    total = sum(score for score in scores if score <= X)
    
    print(total)

if __name__ == '__main__':
    main()