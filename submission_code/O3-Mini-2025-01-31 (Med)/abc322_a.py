def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    S = input_data[1]
    
    # Find first occurrence of 'ABC'
    index = S.find("ABC")
    # Position is 1-indexed, if index is found, output index + 1 else -1
    if index != -1:
        print(index + 1)
    else:
        print(-1)

if __name__ == '__main__':
    main()