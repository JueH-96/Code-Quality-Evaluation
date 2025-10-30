S = input()
pos1 = S.find('|')
pos2 = S.find('|', pos1 + 1)
result = S[0:pos1] + S[pos2 + 1:]
print(result)