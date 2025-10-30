def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    c1 = input_data[1]
    c2 = input_data[2]
    S = input_data[3].strip()
    
    # Generate the new string where every char not equal to c1 is replaced by c2
    result = ''.join(char if char == c1 else c2 for char in S)
    print(result)

if __name__ == '__main__':
    main()