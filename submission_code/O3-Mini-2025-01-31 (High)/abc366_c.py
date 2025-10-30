def main():
    import sys
    input = sys.stdin.readline

    Q = int(input())
    freq = {}
    output_lines = []
    
    for _ in range(Q):
        query = input().split()
        if query[0] == '1':
            # Add one ball with integer x
            x = int(query[1])
            freq[x] = freq.get(x, 0) + 1
        elif query[0] == '2':
            # Remove one ball with integer x (it's guaranteed that such ball exists)
            x = int(query[1])
            freq[x] -= 1
            if freq[x] == 0:
                del freq[x]
        elif query[0] == '3':
            # Print the number of different integers (distinct values)
            output_lines.append(str(len(freq)))
            
    sys.stdout.write("
".join(output_lines))

if __name__ == '__main__':
    main()