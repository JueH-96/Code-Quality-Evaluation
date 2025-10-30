def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    X = int(input_data[1])
    scores = list(map(int, input_data[2:]))
    
    total = sum(score for score in scores if score <= X)
    print(total)

if __name__ == "__main__":
    main()