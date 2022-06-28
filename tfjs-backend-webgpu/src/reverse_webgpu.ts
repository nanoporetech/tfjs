/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class ReverseProgram implements WebGPUProgram {
  variableNames = ['x'];
  workGroupSize: [number, number, number] = [64, 1, 1];
  size = true;
  outputShape: number[];
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  userCode: string;
  rank: number;
  shaderKey: string;
  axis: number[];

  constructor(xShape: number[], axis: number[]) {
    this.outputShape = xShape;
    this.axis = axis;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.rank = xShape.length;
    this.shaderKey = "reverse";

    if (this.rank > 4) {
      throw new Error(
        `WebGPU backend: Reverse of rank-${this.rank} tensor is not yet supported`
      );
    }
  }

  getUserCode(): string {
    const getInCoord = (v: number, i: number) => {
      if (this.axis.indexOf(i) !== -1 && v !== 1) {
        return `${v} - coords[${i}] - 1`;
      }
      return `coords[${i}]`;
    };

    let inCoords;
    if (this.rank === 1) {
      inCoords = `${this.outputShape[0]} - coords - 1`;
    } else {
      inCoords = this.outputShape.map((v, i) => getInCoord(v, i)).join(",");
    }

    return `
          ${getMainHeaderAndGlobalIndexString()}
            if (index < uniforms.size) {
              var coords = getCoordsFromIndex(index);
              setOutputAtIndex(index, getX(${inCoords}));
            }
          }
        `;
  }
}
