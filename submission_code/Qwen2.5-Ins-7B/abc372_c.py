# YOUR CODE HERE
n, q = map(int, input().split())
s = input()
abc_count = s.count('ABC')
abc_end = s.count('AB')

for _ in range(q):
    xi, ci = input().split()
    xi = int(xi)
    if s[xi-1] == ci:
        if s[xi-1] == 'B' and s[xi] == 'C':
            abc_count += int(s[xi-2] == 'A')
        elif s[xi-1] == 'A' and s[xi] == 'B':
            abc_count -= int(s[xi+1] == 'C')
        elif s[xi-1] == 'C' and s[xi] == 'B':
            abc_count -= int(s[xi-2] == 'A')
    else:
        if s[xi-1] == 'A' and ci == 'B':
            abc_count += abc_end
        elif s[xi-1] == 'B' and ci == 'C':
            abc_count += abc_end
        elif s[xi-1] == 'B' and ci == 'A':
            abc_count -= int(s[xi+1] == 'C')
        elif s[xi-1] == 'C' and ci == 'B':
            abc_count -= int(s[xi-2] == 'A')
        elif s[xi-1] == 'B' and ci == 'A':
            abc_end += 1
        elif s[xi-1] == 'C' and ci == 'A':
            abc_end += 1
        elif s[xi-1] == 'A' and ci == 'C':
            abc_end -= 1
        elif s[xi-1] == 'B' and ci == 'C':
            abc_end -= 1
    print(abc_count)