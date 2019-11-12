from database import db, Chromosome

a = input("Are you sure? y/[n]")
if a == 'y':
    Chromosome.query.delete()
    db.session.commit()
    db.engine.execute("drop table Chromosome")
    db.engine.execute("drop table Variables")
    db.create_all()
    db.session.close()
    print("Deleted all Data")