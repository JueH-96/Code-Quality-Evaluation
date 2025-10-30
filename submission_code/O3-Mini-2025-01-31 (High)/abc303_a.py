def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    S = input_data[1]
    T = input_data[2]
    
    for i in range(N):
        x = S[i]
        y = T[i]
        # Check if they are exactly the same character.
        if x == y:
            continue
        # Check for similar pairs: '1' and 'l'
        if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
            continue
        # Check for similar pairs: '0' and 'o'
        if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
            continue
        # If none of the above conditions hold, then they are not similar.
        sys.stdout.write("No")
        return
    
    sys.stdout.write("Yes")
    
if __name__ == '__main__':
    main()