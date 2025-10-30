s = input()
parts = s.split('|')
segments = parts[1:-1]
lengths = [len(segment) for segment in segments]
output_string = ' '.join(map(str, lengths))
print(output_string)