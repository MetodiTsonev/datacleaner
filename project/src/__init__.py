"""DataCleaner pipeline modules.

Eight stages, one module each, in the order the data flows through them:

    loader -> profile -> detect -> plan -> [SPLIT] -> clean -> anomalies
                                                   -> features -> evaluate

`app.py` is a client of these modules and contains no data-processing logic.
See writing/03-design/01-architecture.md.
"""

__version__ = "0.1.0"
