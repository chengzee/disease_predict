import csv
print("ha")
with open("chicken.csv", "a+") as chickenfile:
    write = csv.writer(chickenfile)
    for n in range(10):
        write.writerow([n, "aloha"])

with open("chicken.csv", "r+") as chickenfile:
    rows = csv.reader(chickenfile)
    for row in rows:
        print(3)
        print(row)