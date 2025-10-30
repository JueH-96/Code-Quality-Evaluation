def main():
    import sys
    import re
    
    S = sys.stdin.readline().strip()
    
    # Check if S matches the pattern "A* B* C*"
    if re.match(r'^A*B*C*$', S):
        print("Yes")
    else:
        print("No")

# Do not remove the call to main()
main()