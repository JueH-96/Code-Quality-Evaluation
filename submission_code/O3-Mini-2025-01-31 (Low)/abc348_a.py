def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    result = []
    for i in range(1, N + 1):
        if i % 3 == 0:
            result.append("x")
        else:
            result.append("o")
    print("".join(result))
    
if __name__ == "__main__":
    main()