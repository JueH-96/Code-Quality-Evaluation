import sys

def solve():
    # Read the input line
    line = sys.stdin.readline().split()
    
    # The first word is the surname (S), the second is the first name (T)
    s = line[0]
    # We don't need T for the output
    
    # Construct the output string: surname + space + "san"
    output = s + " " + "san"
    
    # Print the output string
    print(output)

if __name__ == "__main__":
    solve()