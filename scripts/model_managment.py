import mlflow
import pandas as pd
import time
import mlflow.pyfunc
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
class ModelManagment:

    def init_mlflow(experiment_name="exp_1"):
        """
        This function handles the creation of experiment and a run session at
        a specified folder location.
        Args:
            tracking_path(str): the absolute path of the folder to save the traking data
            experiment_name: the name of the experiment.
        
        Returns: 
            experiment_id(string): the id of the created experiment.
        """
        mlflow.end_run()
        # set the tracking uri
        # create the experiment with artifact path
        try:
            # try to create the experiment if it is new
            experiment_id = mlflow.create_experiment(
                            name=experiment_name )
        except:
            # try to load existing experiment if it has been already created
            experiment_id = mlflow.get_experiment_by_name(
                            name=experiment_name).experiment_id
        # set experiment active.
        result = mlflow.set_experiment(experiment_name)

        # create the mlflow_run
        # this run is going to remain open until it is closed by running mlflow.end_run()
        mlflow_run = mlflow.start_run(experiment_id=experiment_id)
        run_id = mlflow_run.info.run_id
        print(result)
        return [experiment_id, run_id]


    def mlflow_saver(model, params, metrics, artifacts=None):
        """
        this function saves the provided sklearn model and associated info and artifacts
        Args:
            model(obj): the sklearn model
            params(dict): parameters used to train the model
            metrics(dict): variables used to check the accuracy and other metrics of the model

        Returns: None. 
        """
        # Track parameters
        mlflow.log_params(params)

        # Track metrics
        mlflow.log_metrics(metrics)

        # Track model
        mlflow.sklearn.log_model(model, "models")

        # Track artifacts
        if artifacts is not None:
            mlflow.log_artifacts(artifacts)

        # closes the opend run
        mlflow.end_run()


    def load_run_info(experiment_name="mlflow-demo"):
        """
        This function loads all saved runs in a given experiment
        and associated paramters and metrics
        Args:
            experiment_name(str): the name of the experiment
        
        Returns:
            data(obj): dataframe containing the info.
        """
        client = MlflowClient()

        # Retrieve Experiment information
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

        # Retrieve Runs information
        ALL_RUNS_INFO = client.list_run_infos(experiment_id)
        ALL_RUNS_ID = [run.run_id for run in ALL_RUNS_INFO]
        ALL_PARAM = [client.get_run(run_id).data.params for run_id in ALL_RUNS_ID]
        ALL_METRIC = [client.get_run(run_id).data.metrics for run_id in ALL_RUNS_ID]
        data = pd.DataFrame()
        data["ids"] = ALL_RUNS_ID
        data["params"] = ALL_PARAM
        data["metrics"] = ALL_METRIC
        return data
        
    def load_model(experiment_name="mlflow-demo", run_id=None):
        """
        This function loads either the most accurate model or the specified model
        if run_id is None, then the most accurate model will be loaded. Otherwise,
        the model with the specified run_id will be loaded.
        Args:
            experiment_name(str): the name of the experiment
            run_id(str): the run id of the model to load
        
        Returns:
            model(obj): sklearn model.
        """
        client = MlflowClient()
        # Retrieve Experiment information
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        if run_id is None:
            data = self.load_run_info(experiment_name)
            data = data.loc[data['metrics'] != {}]
            data["metric_bare"] = data["metrics"].apply(lambda x: x['accuracy'])

            best_run_id = data.sort_values("metric_bare", ascending=False).iloc[0]["ids"]
            
            best_model_path = client.download_artifacts(best_run_id, "models")
            model = mlflow.sklearn.load_model(best_model_path)
        else:        
            selected_model_path = client.download_artifacts(run_id, "models")
            model = mlflow.sklearn.load_model(selected_model_path)

        return model


    def load_model_from_register(model_name, version_stage=None):
        """
        this loads the model from register
        Args:
            model_name(str): the name of the model
            version(int): the version number of the model

        Returns:
            model(obj): loaded sklearn model

        """
        client = MlflowClient()
        if version_stage is not None:
            model_version_uri = f"models:/{model_name}/{version_stage}"
        else:
            version_stage = client.get_latest_versions(model_name)[0].version
            model_version_uri = f"models:/{model_name}/{version_stage}"

        print(f"Loading registered model version from URI: '{model_version_uri}'")
        model = mlflow.pyfunc.load_model(model_version_uri) 

        return model   
    

    def show_stage(model_name, version=None):
        """
        shows the current stage of the model
        Args:
            model_name(str): the name of the model
            version(int): the version number of the model.
        
        Returns: None
        """
        client = MlflowClient()
        if version is not None:
            return client.get_model_version(model_name,version).current_stage
        else:
            version = client.get_latest_versions(model_name)[0].version
            return client.get_model_version(model_name,version).current_stage

        
    def transition_model_stage(model_name, version=None, stage="staging"):
        """
        This function sets the stage of the specified model in registry
        Args:
            model_name(str): the name of the model
            version(int): the version number of the model
            stage(str): the stage of the model (Production, Staging, Archived, or None)

        Returns: None
        """
        client = MlflowClient()
        if version is not None:
            client.transition_model_version_stage(
                            name=model_name,
                            version=version,
                            stage=stage,
                            )
        else:
            version = client.get_latest_versions(model_name)[0].version
            client.transition_model_version_stage(
                            name=model_name,
                            version=version,
                            stage=stage,
                            )


    def register_model(run_id, model_name, sqlite_db = "mydb.sqlite", desc="no description"):
        """
        This function registers the model in the mlflow registry
        Args:
            run_id(str): run id of the model to be registered
            model_name(str): the name of the model to be used when registering
            sqlite_db(str): the name of the database file to be used as register
            desc(str): description of the model
        
        Returns:
            model_detials(obj): registered model details. 
        """
        mlflow.set_registry_uri(f"sqlite:///{sqlite_db}")

        artifact_path = "models"
        model_uri = "runs:/{run_id}/models".format(run_id=run_id)

        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

        client = MlflowClient()
        for _ in range(10):
            model_version_details = client.get_model_version(
            name=model_details.name,
            version=model_details.version,
            )
            status = ModelVersionStatus.from_string(model_version_details.status)
            print("Model status: %s" % ModelVersionStatus.to_string(status))
            if status == ModelVersionStatus.READY:
                break
            time.sleep(1)
    
        client.update_model_version(
                                name=model_details.name,
                                version=model_details.version,
                                description=desc
                                )
        
        return model_details
    
    def log_model(artifact_name, model):
        """
        A function that logs necessary information of the model.
        
        Args:
            artifact_name: name to save the model artifact  with
            model: the model
            
        Returns:
            model_uri
        """
        with mlflow.start_run():
            mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_name)
            run_id = mlflow.active_run().info.run_id
            model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run_id, artifact_path=artifact_name)
            
        return model_uri
