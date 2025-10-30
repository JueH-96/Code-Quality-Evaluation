def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    L = int(input_data[1])
    scores = list(map(int, input_data[2:2+N]))
    
    count = sum(1 for score in scores if score >= L)
    print(count)

if __name__ == '__main__':
    main()