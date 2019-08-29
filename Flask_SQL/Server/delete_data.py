from database import *

a = input("Are you sure? y/[n]")
if a == 'y':
    Chromosome.query.delete()
    db.session.commit()
    db.session.close()
    print("Deleted all Data")