def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    s = input_data[1]
    
    index = s.find("ABC")
    if index == -1:
        print(-1)
    else:
        # add 1 since output expects 1-indexed position
        print(index + 1)

if __name__ == "__main__":
    main()