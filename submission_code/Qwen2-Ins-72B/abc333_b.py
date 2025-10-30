S1,S2 = input()
T1,T2 = input()

if sorted([S1,S2]) == sorted([T1,T2]):
  print("Yes")
elif sorted([S1,S2]) == ["A","C"] and sorted([T1,T2]) == ["C","E"]:
  print("Yes")
elif sorted([S1,S2]) == ["C","E"] and sorted([T1,T2]) == ["A","C"]:
  print("Yes")
elif sorted([S1,S2]) == ["A","D"] and sorted([T1,T2]) == ["B","E"]:
  print("Yes")
elif sorted([S1,S2]) == ["B","E"] and sorted([T1,T2]) == ["A","D"]:
  print("Yes")
else:
  print("No")