# nordic_model

This is an open dispatch model for the north European power system. More details of the model are provided in [1]. If you use the whole or parts of the model, please cite the paper. 

USE OF MODEL

To use the model, perform the following steps:
1. Download github directory.
2. Download the database files provided in db_files.zip: https://drive.google.com/file/d/1X7BV3OnyBE73_1TE4n-igFE7w2eCSX9M/view?usp=sharing.
   Extract the database files (e.g in the /Data folder), this path is now db_path.
3. Obtain a personal access token for the ENTSO-E Transparency Platform. See the instructions at https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html. 
   Copy your access token into token.txt.
4. Run create_databases() from create_databases.py. You must specify db_path as the input parameter. Downloading the necessary data will take approximately 10 hours for unit.db, 5 hours for gen.db, and less than 1 hour for the remaining databases. If you want to shorten the time, omit downloading unit.db which is not normally used, or reduce the range of years for which data is downloaded (2016-2020 by default). Note that you should copy the exisiting .db files into db_path (Step 2) before you run create_databases(). This is because gen.db already contains data for Sweden, which does not exist on the Transparency platform.
5. You are now ready to use the model. See run_example.py for an example.

MODEL OPTIONS 

The model options are documented in the code, in default_options() in the Model class.


REFERENCES:

[1] E. Nycander and L. Söder, "An Open Dispatch Model for the Nordic Power System", Preprint, 2021, doi: https://doi.org/10.13140/RG.2.2.31948.74884/1.