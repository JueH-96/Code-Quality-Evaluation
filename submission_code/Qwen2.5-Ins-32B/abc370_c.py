# YOUR CODE HERE
import sys

def transform_string(s, t):
    if s == t:
        return []
    
    steps = []
    for i in range(len(s)):
        if s[i] != t[i]:
            new_s = s[:i] + t[i] + s[i+1:]
            steps.append(new_s)
            s = new_s
    
    return steps

def main():
    s = input().strip()
    t = input().strip()
    
    steps = transform_string(s, t)
    print(len(steps))
    for step in steps:
        print(step)

if __name__ == "__main__":
    main()