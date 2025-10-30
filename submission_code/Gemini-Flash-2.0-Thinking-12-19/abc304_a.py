import sys

def solve():
    n = int(sys.stdin.readline())
    names = []
    ages = []
    for _ in range(n):
        line = sys.stdin.readline().split()
        name = line[0]
        age = int(line[1])
        names.append(name)
        ages.append(age)
    
    youngest_index = 0
    min_age = ages[0]
    for i in range(1, n):
        if ages[i] < min_age:
            min_age = ages[i]
            youngest_index = i
            
    output_names = []
    for i in range(n):
        current_index = (youngest_index + i) % n
        output_names.append(names[current_index])
        
    for name in output_names:
        print(name)

if __name__ == '__main__':
    solve()