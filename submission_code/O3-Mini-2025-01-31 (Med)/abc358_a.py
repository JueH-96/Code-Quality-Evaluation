def main():
    import sys
    input_line = sys.stdin.read().strip()
    if input_line:
        parts = input_line.split()
        if len(parts) == 2:
            S, T = parts
            if S == "AtCoder" and T == "Land":
                print("Yes")
                return
    print("No")
    
if __name__ == '__main__':
    main()