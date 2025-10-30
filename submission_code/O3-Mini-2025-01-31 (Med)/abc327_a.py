def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    s = input_data[1]
    
    if "ab" in s or "ba" in s:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()