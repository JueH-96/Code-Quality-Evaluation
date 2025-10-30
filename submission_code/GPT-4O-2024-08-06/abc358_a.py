# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    S, T = data.split()
    
    if S == "AtCoder" and T == "Land":
        print("Yes")
    else:
        print("No")

main()