from engine.trainer import TrainerBuilder

if __name__ == '__main__':
    mdict = {"name": "mlp", "in_features":10, "out_features":2, "hidden_features":16, "bias": True, "reg": {"name": "L2", "alpha": 0.5}}
    odict = {"name": "sgd", "lr":0.01, "momentum":0.9, "weight_decay":0.00001, "precision": "float32"}
    ddict = {"type": "stream","svc_url": "http://192.168.56.20:8094", "table_name": "frappe_train", "namespace": "train", "columns": ["label", "col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"], "batch_size": 16}
    tdict = {"device": "cpu", "seed": 0, "max_epoch": 1000}


    builder = TrainerBuilder()
    trainer = (builder.build_name("trainer1")
               .build_model(mdict)
               .build_optimizer(odict)
               .build_train_dataloader(ddict)
               .build_train_config(tdict)
               .build())

    trainer.train()
    print(trainer.model.get_params())
    print("done")