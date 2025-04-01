# PyCIL Toolbox Error Collection:

## General:
### #1: Related to SVHN dataset
    2025-03-25 14:54:42,775 [trainer.py] => config: ./exps/fostersvhn.json
    2025-03-25 14:54:42,776 [trainer.py] => prefix: cil
    2025-03-25 14:54:42,776 [trainer.py] => dataset: svhn
    2025-03-25 14:54:42,776 [trainer.py] => memory_size: 2000
    2025-03-25 14:54:42,776 [trainer.py] => memory_per_class: 20
    2025-03-25 14:54:42,776 [trainer.py] => fixed_memory: True
    2025-03-25 14:54:42,776 [trainer.py] => shuffle: False
    2025-03-25 14:54:42,776 [trainer.py] => init_cls: 2
    2025-03-25 14:54:42,776 [trainer.py] => increment: 2
    2025-03-25 14:54:42,776 [trainer.py] => model_name: foster
    2025-03-25 14:54:42,776 [trainer.py] => convnet_type: resnet32
    2025-03-25 14:54:42,776 [trainer.py] => device: [device(type='cuda', index=0)]
    2025-03-25 14:54:42,776 [trainer.py] => seed: 1993
    2025-03-25 14:54:42,777 [trainer.py] => beta1: 0.96
    2025-03-25 14:54:42,777 [trainer.py] => beta2: 0.97
    2025-03-25 14:54:42,777 [trainer.py] => oofc: ft
    2025-03-25 14:54:42,777 [trainer.py] => is_teacher_wa: False
    2025-03-25 14:54:42,777 [trainer.py] => is_student_wa: False
    2025-03-25 14:54:42,777 [trainer.py] => lambda_okd: 1
    2025-03-25 14:54:42,777 [trainer.py] => wa_value: 1
    2025-03-25 14:54:42,777 [trainer.py] => init_epochs: 1
    2025-03-25 14:54:42,777 [trainer.py] => init_lr: 0.1
    2025-03-25 14:54:42,777 [trainer.py] => init_weight_decay: 0.0005
    2025-03-25 14:54:42,777 [trainer.py] => boosting_epochs: 1
    2025-03-25 14:54:42,777 [trainer.py] => compression_epochs: 1
    2025-03-25 14:54:42,777 [trainer.py] => lr: 0.1
    2025-03-25 14:54:42,777 [trainer.py] => batch_size: 128
    2025-03-25 14:54:42,777 [trainer.py] => weight_decay: 0.0005
    2025-03-25 14:54:42,778 [trainer.py] => num_workers: 8
    2025-03-25 14:54:42,778 [trainer.py] => T: 2
    100% 182M/182M [01:02<00:00, 2.91MB/s]
    100% 64.3M/64.3M [00:34<00:00, 1.86MB/s]
    2025-03-25 14:56:25,464 [data_manager.py] => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    2025-03-25 14:56:25,568 [trainer.py] => All params: 0
    2025-03-25 14:56:25,569 [trainer.py] => Trainable params: 0
    2025-03-25 14:56:25,589 [foster.py] => Learning on 0-2
    2025-03-25 14:56:25,589 [foster.py] => All params: 464414
    2025-03-25 14:56:25,590 [foster.py] => Trainable params: 464414
    /usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
      0% 0/1 [00:00<?, ?it/s]
    Traceback (most recent call last):
      File "/content/PyCIL-Saliency/main.py", line 31, in <module>
        main()
      File "/content/PyCIL-Saliency/main.py", line 12, in main
        train(args)
      File "/content/PyCIL-Saliency/trainer.py", line 19, in train
        _train(args)
      File "/content/PyCIL-Saliency/trainer.py", line 83, in _train
        model.incremental_train(data_manager)
      File "/content/PyCIL-Saliency/models/foster.py", line 87, in incremental_train
        self._train(self.train_loader, self.test_loader)
      File "/content/PyCIL-Saliency/models/foster.py", line 112, in _train
        self._init_train(train_loader, test_loader, optimizer, scheduler)
      File "/content/PyCIL-Saliency/models/foster.py", line 175, in _init_train
        for i, (_, inputs, targets) in enumerate(train_loader):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 708, in __next__
        data = self._next_data()
               ^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1480, in _next_data
        return self._process_data(data)
               ^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1505, in _process_data
        data.reraise()
      File "/usr/local/lib/python3.11/dist-packages/torch/_utils.py", line 733, in reraise
        raise exception
    TypeError: Caught TypeError in DataLoader worker process 0.
    Original Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3311, in fromarray
        mode, rawmode = _fromarray_typemap[typekey]
                        ~~~~~~~~~~~~~~~~~~^^^^^^^^^
    KeyError: ((1, 1, 32), '|u1')
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
        data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
               ^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
        data = [self.dataset[idx] for idx in possibly_batched_index]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
        data = [self.dataset[idx] for idx in possibly_batched_index]
                ~~~~~~~~~~~~^^^^^
      File "/content/PyCIL-Saliency/utils/data_manager.py", line 260, in __getitem__
        image = self.trsf(Image.fromarray(self.images[idx]))
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3315, in fromarray
        raise TypeError(msg) from e
    TypeError: Cannot handle this data type: (1, 1, 32), |u1


## Foster:
### #1:
    Traceback (most recent call last):
      File "/coe_data/AI_CL/PyCIL/main.py", line 31, in <module>
        main()
      File "/coe_data/AI_CL/PyCIL/main.py", line 12, in main
        train(args)
      File "/coe_data/AI_CL/PyCIL/trainer.py", line 19, in train
        _train(args)
      File "/coe_data/AI_CL/PyCIL/trainer.py", line 126, in _train
        np_acctable[idxx, :idxy] = np.array(line)
    ValueError: could not broadcast input array from shape (7,) into shape (6,)
Never encountered again.