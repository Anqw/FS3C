# Base datasets

For a few datasets that Fs3c natively supports,
the datasets are assumed to exist in a directory called
"datasets/", under the directory where you launch the program.
They need to have the following directory structure:

## Expected dataset structure for Pascal VOC:
```
VOC20{07,12}/
  Annotations/
  ImageSets/
  JPEGImages/
```

## Expected dataset structure for COCO:
```
coco/
  annotations/
    instances_{train,val}2014.json
  trainval2014/
    # image files that are mentioned in the corresponding json
```


# Few-shot datasets

## Pascal VOC:
```
vocsplit/
  box_{1,2,3,5,10}shot_{category}_train.txt
  seed{1-29}/
    # shots
```

Dataset names for config files:
```
voc_20{07,12}_trainval_{base,all}{1,2,3}        # Train/val datasets with base categories or all
                                                  categories for splits 1, 2, and 3.
voc_2007_trainval_all{1,2,3}_{1,2,3,5,10}shot   # Balanced subsets for splits 1, 2, and 3 containing
                                                  1, 2, 3, 5, or 10 shots for each category. You only
                                                  need to specify 2007, as it will load in both 2007
                                                  and 2012 automatically.
voc_2007_trainval_novel{1,2,3}_{1,2,3,5,10}shot # Same as previous datasets, but only contains data
                                                  of novel categories.
voc_2007_test_{base,novel,all}{1,2,3}           # Test datasets with base categories, novel categories,
                                                  or all categories for splits 1, 2, and 3.
```

## COCO:
```
cocosplit/
  datasplit/
    trainvalno5k.json
    5k.json
  full_box_{1,2,3,5,10,30}shot_{category}_trainval.json
  seed{1-9}/
    # shots
```

Dataset names for config files:
```
coco_trainval_{base,all}                        # Train/val datasets with base categories or all
                                                  categories.
coco_trainval_all_{1,2,3,5,10,30}shot           # Balanced subsets containing 1, 2, 3, 5, 10, or 30
                                                  shots for each category.
coco_trainval_novel_{1,2,3,5,10,30}shot         # Same as previous datasets, but only contains data
                                                  of novel categories.
coco_test_{base,novel,all}                      # Test datasets with base categories, novel categories,
                                                  or all categories.
```