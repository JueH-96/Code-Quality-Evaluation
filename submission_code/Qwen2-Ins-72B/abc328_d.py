S = input()

while True:
    idx = S.find('ABC')
    if idx == -1:
        break
    S = S[:idx] + S[idx+3:]
    
print(S)