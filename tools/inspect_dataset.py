#!/usr/bin/env python3
"""Read-only WeedyRice integrity audit; full scan intentionally reads every pixel."""
import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/root/train/dataset/weedy-rice/extracted/WeedyRice-RGBMS-DB')
BANDS = ('G', 'R', 'RE', 'NIR')

def indexed_files(directory):
    result = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        key = path.name.casefold()
        if key in result:
            raise ValueError(f'duplicate case-insensitive filename: {result[key]} and {path}')
        result[key] = path
    return result

def check_band(path):
    with Image.open(path) as image:
        a = np.asarray(image)
        return {'size': list(image.size), 'shape': list(a.shape), 'dtype': str(a.dtype),
                'channels_identical': bool(a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 3 and np.array_equal(a[:,:,0], a[:,:,1]) and np.array_equal(a[:,:,0], a[:,:,2])))}

def split_intersections(splits):
    return {f'{a}:{b}': sorted(set(x.casefold() for x in splits[a]) & set(x.casefold() for x in splits[b])) for a,b in itertools.combinations(splits, 2)}

def band_matches(band, metadata_band):
    expected = {"G":"green", "R":"red", "RE":"rededge", "NIR":"nir"}
    return metadata_band.casefold().replace(" ", "") == expected[band]

def audit(root):
    dirs = {d: indexed_files(root/d) for d in ('RGB','Masks','Multispectral')}
    splits = {s: [x.strip() for x in (root/f'{s}_list.txt').read_text().splitlines() if x.strip()] for s in ('train','val','test')}
    report = {'file_counts': {k:len(v) for k,v in dirs.items()}, 'splits': {}, 'intersections': split_intersections(splits), 'size_counts': {}, 'mask_values': {}, 'band_shapes': {}, 'band_dtypes': {}, 'non_identical_bands': [], 'size_mismatches': [], 'missing': [], 'metadata_errors': [], 'read_errors': [], 'alignment_note': 'Equal dimensions and matching names do not establish pixel-level spatial registration.'}
    all_ids = []
    for split, names in splits.items():
        keys = [x.casefold() for x in names]
        duplicates = [x for x,n in Counter(keys).items() if n>1]
        report['splits'][split] = {'listed':len(names), 'duplicates':duplicates, 'paired':0, 'foreground_pixels':0, 'pixels':0}
        for name in names:
            stem = Path(name).stem
            keys_needed = [('RGB',name), ('Masks',stem+'.png')] + [('Multispectral',stem+'_'+b+'.tif') for b in BANDS]
            missing = [f'{d}/{f}' for d,f in keys_needed if f.casefold() not in dirs[d]]
            if missing:
                report['missing'].extend(missing); continue
            report['splits'][split]['paired'] += 1
            all_ids.append((split, name, stem))
    size_counts, values, shapes, dtypes = Counter(), Counter(), Counter(), Counter()
    for split,name,stem in all_ids:
        try:
            with Image.open(dirs['RGB'][name.casefold()]) as im:
                rgb_size = im.size
            size_counts[str(rgb_size)] += 1
            with Image.open(dirs['Masks'][(stem+'.png').casefold()]) as im:
                mask = np.asarray(im)
                if im.size != rgb_size: report['size_mismatches'].append(str(im.filename))
                unique, counts = np.unique(mask, return_counts=True)
                values.update({str(int(v)):int(c) for v,c in zip(unique,counts)})
                report['splits'][split]['foreground_pixels'] += int((mask != 0).sum())
                report['splits'][split]['pixels'] += int(mask.size)
            for band in BANDS:
                path = dirs['Multispectral'][(stem+'_'+band+'.tif').casefold()]
                info = check_band(path)
                shapes[str(info['shape'])] += 1; dtypes[info['dtype']] += 1
                if tuple(info['size']) != rgb_size: report['size_mismatches'].append(str(path))
                if not info['channels_identical']: report['non_identical_bands'].append(str(path))
        except Exception as exc:
            report['read_errors'].append({'sample':name, 'error':repr(exc)})
    report['size_counts'], report['mask_values'], report['band_shapes'], report['band_dtypes'] = map(dict,(size_counts, values, shapes, dtypes))
    for s in report['splits'].values():
        s['foreground_fraction'] = s['foreground_pixels']/s['pixels'] if s['pixels'] else None
    with (root/'Metadata'/'filename_mapping.csv').open(newline='') as f: mapping = list(csv.DictReader(f))
    with (root/'Metadata'/'image_metadata.csv').open(newline='') as f: metadata = list(csv.DictReader(f))
    orig = {x['original_filename']:x for x in metadata}
    if len(orig) != len(metadata): report['metadata_errors'].append('duplicate original filename in image_metadata.csv')
    standardized = {}
    for row in mapping:
        key = row['standardized_filename'].casefold()
        if key in standardized: report['metadata_errors'].append('duplicate standardized mapping: '+key)
        standardized[key] = row['original_filename']
        if row['original_filename'] not in orig: report['metadata_errors'].append('metadata missing: '+row['original_filename'])
    dates = {s:set() for s in splits}
    groups = {}
    for split,name,stem in all_ids:
        for filename in [name]+[stem+'_'+b+'.TIF' for b in BANDS]:
            original = standardized.get(filename.casefold())
            if not original: report['metadata_errors'].append('mapping missing: '+filename); continue
            row = orig.get(original)
            if not row: continue
            band = None if filename == name else Path(filename).stem.rsplit('_',1)[-1]
            if (band is None and row['band']) or (band is not None and not band_matches(band, row['band'])): report['metadata_errors'].append('band mismatch: '+filename)
            if filename == name:
                dates[split].add(row['acquisition_date'])
                parts = original.split('_')
                if len(parts)>=3 and parts[2].isdigit(): groups.setdefault(row['acquisition_date'], []).append((int(parts[2]), split, name))
    report['shared_acquisition_dates'] = {f'{a}:{b}':sorted(dates[a]&dates[b]) for a,b in itertools.combinations(splits,2)}
    adjacent=[]
    for date, entries in groups.items():
        entries.sort()
        for x,y in zip(entries, entries[1:]):
            if y[0]-x[0]==1 and x[1]!=y[1]: adjacent.append({'date':date,'samples':[x[2],y[2]],'splits':[x[1],y[1]]})
    report['adjacent_cross_split_count']=len(adjacent); report['adjacent_cross_split_examples']=adjacent[:10]
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=Path('results/dataset_audit.json'))
    args=parser.parse_args(); report=audit(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if any(report[k] for k in ('missing','size_mismatches','non_identical_bands','metadata_errors','read_errors')) or any(report['intersections'].values()) or any(v['duplicates'] for v in report['splits'].values()):
        raise SystemExit(1)

if __name__ == '__main__': main()
